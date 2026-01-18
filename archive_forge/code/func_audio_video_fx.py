import decorator
from moviepy.tools import cvsecs
@decorator.decorator
def audio_video_fx(f, clip, *a, **k):
    """ Use an audio function on a video/audio clip
    
    This decorator tells that the function f (audioclip -> audioclip)
    can be also used on a video clip, at which case it returns a
    videoclip with unmodified video and modified audio.
    """
    if hasattr(clip, 'audio'):
        newclip = clip.copy()
        if clip.audio is not None:
            newclip.audio = f(clip.audio, *a, **k)
        return newclip
    else:
        return f(clip, *a, **k)