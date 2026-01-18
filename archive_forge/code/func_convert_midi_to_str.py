import json
import random
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, floor, log
from typing import Dict, Iterator, List, Optional, Tuple
import mido
def convert_midi_to_str(cfg: VocabConfig, filter_cfg: FilterConfig, mid: mido.MidiFile, augment: AugmentValues=None) -> List[str]:
    utils = VocabUtils(cfg)
    if augment is None:
        augment = AugmentValues.default()
    for i in range(len(mid.tracks)):
        mid.tracks[i] = [msg for msg in mid.tracks[i] if msg.type != 'unknown_meta']
    if len(mid.tracks) > 1:
        mid.tracks = [mido.merge_tracks(mid.tracks)]
    delta_time_ms = 0.0
    tempo = 500000
    channel_program = {i: 0 for i in range(16)}
    channel_volume = {i: 127 for i in range(16)}
    channel_expression = {i: 127 for i in range(16)}
    channel_notes = {i: {} for i in range(16)}
    channel_pedal_on = {i: False for i in range(16)}
    channel_pedal_events = {i: {} for i in range(16)}
    started_flag = False
    output_list = []
    output = ['<start>']
    output_length_ms = 0.0
    token_data_buffer: List[Tuple[int, int, int, float]] = []

    def flush_token_data_buffer():
        nonlocal token_data_buffer, output, cfg, utils, augment
        token_data = [x for x in utils.prog_data_list_to_token_data_list(token_data_buffer)]
        if augment.instrument_bin_remap or augment.transpose_semitones:
            raw_transpose = lambda bin, n: n + augment.transpose_semitones if bin != cfg._ch10_bin_int else n
            octave_shift_if_oob = lambda n: n + 12 if n < 0 else n - 12 if n >= cfg.note_events else n
            transpose = lambda bin, n: octave_shift_if_oob(raw_transpose(bin, n))
            token_data = [(augment.instrument_bin_remap.get(i, i), transpose(i, n), v) for i, n, v in token_data]
        if cfg.do_token_sorting:
            token_data = utils.sort_token_data(token_data)
        if cfg.unrolled_tokens:
            for t in token_data:
                output += [utils.format_unrolled_instrument_bin(t[0]), utils.format_unrolled_note(t[1]), utils.format_unrolled_velocity(t[2])]
        else:
            output += [utils.format_note_token(*t) for t in token_data]
        token_data_buffer = []

    def consume_note_program_data(prog: int, chan: int, note: int, vel: float):
        nonlocal output, output_length_ms, started_flag, delta_time_ms, cfg, utils, token_data_buffer
        is_token_valid = utils.prog_data_to_token_data(prog, chan, note, vel) is not None
        if not is_token_valid:
            return
        if delta_time_ms > filter_cfg.piece_split_delay * 1000.0:
            silent = True
            for channel in channel_notes.keys():
                if len(channel_notes[channel]) > 0:
                    silent = False
                    break
            if silent:
                flush_token_data_buffer()
                output.append('<end>')
                if output_length_ms > filter_cfg.min_piece_length * 1000.0:
                    output_list.append(' '.join(output))
                output = ['<start>']
                output_length_ms = 0.0
                started_flag = False
        if started_flag:
            wait_tokens = utils.data_to_wait_tokens(delta_time_ms)
            if len(wait_tokens) > 0:
                flush_token_data_buffer()
                output_length_ms += delta_time_ms
                output += wait_tokens
        delta_time_ms = 0.0
        token_data_buffer.append((prog, chan, note, vel * augment.velocity_mod_factor))
        started_flag = True
    for msg in mid.tracks[0]:
        time_ms = mido.tick2second(msg.time, mid.ticks_per_beat, tempo) * 1000.0
        delta_time_ms += time_ms
        t = msg.type
        if msg.is_meta:
            if t == 'set_tempo':
                tempo = msg.tempo * augment.time_stretch_factor
            continue

        def handle_note_off(ch, prog, n):
            if channel_pedal_on[ch]:
                channel_pedal_events[ch][n, prog] = True
            else:
                consume_note_program_data(prog, ch, n, 0)
                if n in channel_notes[ch]:
                    del channel_notes[ch][n]
        if t == 'program_change':
            channel_program[msg.channel] = msg.program
        elif t == 'note_on':
            if msg.velocity == 0:
                handle_note_off(msg.channel, channel_program[msg.channel], msg.note)
            else:
                if (msg.note, channel_program[msg.channel]) in channel_pedal_events[msg.channel]:
                    del channel_pedal_events[msg.channel][msg.note, channel_program[msg.channel]]
                consume_note_program_data(channel_program[msg.channel], msg.channel, msg.note, mix_volume(msg.velocity, channel_volume[msg.channel], channel_expression[msg.channel]))
                channel_notes[msg.channel][msg.note] = True
        elif t == 'note_off':
            handle_note_off(msg.channel, channel_program[msg.channel], msg.note)
        elif t == 'control_change':
            if msg.control == 7 or msg.control == 39:
                channel_volume[msg.channel] = msg.value
            elif msg.control == 11:
                channel_expression[msg.channel] = msg.value
            elif msg.control == 64:
                channel_pedal_on[msg.channel] = msg.value >= 64
                if not channel_pedal_on[msg.channel]:
                    for note, program in channel_pedal_events[msg.channel]:
                        handle_note_off(msg.channel, program, note)
                    channel_pedal_events[msg.channel] = {}
            elif msg.control == 123:
                for channel in channel_notes.keys():
                    for note in list(channel_notes[channel]).copy():
                        handle_note_off(channel, channel_program[channel], note)
        else:
            pass
    flush_token_data_buffer()
    output.append('<end>')
    if output_length_ms > filter_cfg.min_piece_length * 1000.0:
        output_list.append(' '.join(output))
    return output_list