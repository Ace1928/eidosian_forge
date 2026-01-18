from midi2audio import FluidSynth
import argparse
import os
import subprocess
def play_midi(self, midi_file):
    subprocess.call(['fluidsynth', '-i', self.sound_font, midi_file, '-r', str(self.sample_rate)])