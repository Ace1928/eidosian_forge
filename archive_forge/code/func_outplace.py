import decorator
from moviepy.tools import cvsecs
@decorator.decorator
def outplace(f, clip, *a, **k):
    """ Applies f(clip.copy(), *a, **k) and returns clip.copy()"""
    newclip = clip.copy()
    f(newclip, *a, **k)
    return newclip