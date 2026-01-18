from __future__ import division
from moviepy.audio.AudioClip import AudioClip
from moviepy.audio.io.readers import FFMPEG_AudioReader
def coreader(self):
    """ Returns a copy of the AudioFileClip, i.e. a new entrance point
            to the audio file. Use copy when you have different clips
            watching the audio file at different times. """
    return AudioFileClip(self.filename, self.buffersize)