import decorator
from moviepy.tools import cvsecs
@decorator.decorator
def apply_to_audio(f, clip, *a, **k):
    """ This decorator will apply the function f to the audio of
        the clip created with f """
    newclip = f(clip, *a, **k)
    if getattr(newclip, 'audio', None):
        newclip.audio = f(newclip.audio, *a, **k)
    return newclip