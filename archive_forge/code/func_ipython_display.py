import os
from base64 import b64encode
from moviepy.audio.AudioClip import AudioClip
from moviepy.tools import extensions_dict
from ..VideoClip import ImageClip, VideoClip
from .ffmpeg_reader import ffmpeg_parse_infos
def ipython_display(clip, filetype=None, maxduration=60, t=None, fps=None, rd_kwargs=None, center=True, **html_kwargs):
    """
    clip
      Either the name of a file, or a clip to preview. The clip will
      actually be written to a file and embedded as if a filename was
      provided.

    filetype:
      One of 'video','image','audio'. If None is given, it is determined
      based on the extension of ``filename``, but this can bug.

    maxduration
      An error will be raised if the clip's duration is more than the indicated
      value (in seconds), to avoid spoiling the  browser's cache and the RAM.

    t
      If not None, only the frame at time t will be displayed in the notebook,
      instead of a video of the clip

    fps
      Enables to specify an fps, as required for clips whose fps is unknown.
    
    **kwargs:
      Allow you to give some options, like width=260, etc. When editing
      looping gifs, a good choice is loop=1, autoplay=1.
    
    Remarks: If your browser doesn't support HTML5, this should warn you.
    If nothing is displayed, maybe your file or filename is wrong.
    Important: The media will be physically embedded in the notebook.

    Examples
    =========

    >>> import moviepy.editor as mpy
    >>> # later ...
    >>> clip.write_videofile("test.mp4")
    >>> mpy.ipython_display("test.mp4", width=360)

    >>> clip.audio.write_audiofile('test.ogg') # Sound !
    >>> mpy.ipython_display('test.ogg')

    >>> clip.write_gif("test.gif")
    >>> mpy.ipython_display('test.gif')

    >>> clip.save_frame("first_frame.jpeg")
    >>> mpy.ipython_display("first_frame.jpeg")
    """
    if not ipython_available:
        raise ImportError('Only works inside an IPython Notebook')
    if rd_kwargs is None:
        rd_kwargs = {}
    if fps is not None:
        rd_kwargs['fps'] = fps
    if t is not None:
        clip = clip.to_ImageClip(t)
    return HTML2(html_embed(clip, filetype=filetype, maxduration=maxduration, center=center, rd_kwargs=rd_kwargs, **html_kwargs))