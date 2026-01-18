import json
import random
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, floor, log
from typing import Dict, Iterator, List, Optional, Tuple
import mido
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