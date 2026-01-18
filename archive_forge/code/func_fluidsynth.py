import numpy as np
import os
import pkg_resources
from .containers import PitchBend
from .utilities import pitch_bend_to_semitones, note_number_to_hz
def fluidsynth(self, fs=44100, sf2_path=None):
    """Synthesize using fluidsynth.

        Parameters
        ----------
        fs : int
            Sampling rate to synthesize.
        sf2_path : str
            Path to a .sf2 file.
            Default ``None``, which uses the TimGM6mb.sf2 file included with
            ``pretty_midi``.

        Returns
        -------
        synthesized : np.ndarray
            Waveform of the MIDI data, synthesized at ``fs``.

        """
    if sf2_path is None:
        sf2_path = pkg_resources.resource_filename(__name__, DEFAULT_SF2)
    if not _HAS_FLUIDSYNTH:
        raise ImportError('fluidsynth() was called but pyfluidsynth is not installed.')
    if not os.path.exists(sf2_path):
        raise ValueError('No soundfont file found at the supplied path {}'.format(sf2_path))
    if len(self.notes) == 0:
        return np.array([])
    fl = fluidsynth.Synth(samplerate=fs)
    sfid = fl.sfload(sf2_path)
    if self.is_drum:
        channel = 9
        res = fl.program_select(channel, sfid, 128, self.program)
        if res == -1:
            fl.program_select(channel, sfid, 128, 0)
    else:
        channel = 0
        fl.program_select(channel, sfid, 0, self.program)
    event_list = []
    for note in self.notes:
        event_list += [[note.start, 'note on', note.pitch, note.velocity]]
        event_list += [[note.end, 'note off', note.pitch]]
    for bend in self.pitch_bends:
        event_list += [[bend.time, 'pitch bend', bend.pitch]]
    for control_change in self.control_changes:
        event_list += [[control_change.time, 'control change', control_change.number, control_change.value]]
    event_list.sort(key=lambda x: (x[0], x[1] != 'note off'))
    current_time = event_list[0][0]
    next_event_times = [e[0] for e in event_list[1:]]
    for event, end in zip(event_list[:-1], next_event_times):
        event[0] = end - event[0]
    event_list[-1][0] = 1.0
    total_time = current_time + np.sum([e[0] for e in event_list])
    synthesized = np.zeros(int(np.ceil(fs * total_time)))
    for event in event_list:
        if event[1] == 'note on':
            fl.noteon(channel, event[2], event[3])
        elif event[1] == 'note off':
            fl.noteoff(channel, event[2])
        elif event[1] == 'pitch bend':
            fl.pitch_bend(channel, event[2])
        elif event[1] == 'control change':
            fl.cc(channel, event[2], event[3])
        current_sample = int(fs * current_time)
        end = int(fs * (current_time + event[0]))
        samples = fl.get_samples(end - current_sample)[::2]
        synthesized[current_sample:end] += samples
        current_time += event[0]
    fl.delete()
    return synthesized