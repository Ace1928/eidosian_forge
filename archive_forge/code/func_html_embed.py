import os
from base64 import b64encode
from moviepy.audio.AudioClip import AudioClip
from moviepy.tools import extensions_dict
from ..VideoClip import ImageClip, VideoClip
from .ffmpeg_reader import ffmpeg_parse_infos
def html_embed(clip, filetype=None, maxduration=60, rd_kwargs=None, center=True, **html_kwargs):
    """ Returns HTML5 code embedding the clip
    
    clip
      Either a file name, or a clip to preview.
      Either an image, a sound or a video. Clips will actually be
      written to a file and embedded as if a filename was provided.


    filetype
      One of 'video','image','audio'. If None is given, it is determined
      based on the extension of ``filename``, but this can bug.
    
    rd_kwargs
      keyword arguments for the rendering, like {'fps':15, 'bitrate':'50k'}
    

    **html_kwargs
      Allow you to give some options, like width=260, autoplay=True,
      loop=1 etc.

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
    if rd_kwargs is None:
        rd_kwargs = {}
    if 'Clip' in str(clip.__class__):
        TEMP_PREFIX = '__temp__'
        if isinstance(clip, ImageClip):
            filename = TEMP_PREFIX + '.png'
            kwargs = {'filename': filename, 'withmask': True}
            kwargs.update(rd_kwargs)
            clip.save_frame(**kwargs)
        elif isinstance(clip, VideoClip):
            filename = TEMP_PREFIX + '.mp4'
            kwargs = {'filename': filename, 'verbose': False, 'preset': 'ultrafast'}
            kwargs.update(rd_kwargs)
            clip.write_videofile(**kwargs)
        elif isinstance(clip, AudioClip):
            filename = TEMP_PREFIX + '.mp3'
            kwargs = {'filename': filename, 'verbose': False}
            kwargs.update(rd_kwargs)
            clip.write_audiofile(**kwargs)
        else:
            raise ValueError('Unknown class for the clip. Cannot embed and preview.')
        return html_embed(filename, maxduration=maxduration, rd_kwargs=rd_kwargs, center=center, **html_kwargs)
    filename = clip
    options = ' '.join(["%s='%s'" % (str(k), str(v)) for k, v in html_kwargs.items()])
    name, ext = os.path.splitext(filename)
    ext = ext[1:]
    if filetype is None:
        ext = filename.split('.')[-1].lower()
        if ext == 'gif':
            filetype = 'image'
        elif ext in extensions_dict:
            filetype = extensions_dict[ext]['type']
        else:
            raise ValueError("No file type is known for the provided file. Please provide argument `filetype` (one of 'image', 'video', 'sound') to the ipython display function.")
    if filetype == 'video':
        exts_htmltype = {'mp4': 'mp4', 'webm': 'webm', 'ogv': 'ogg'}
        allowed_exts = ' '.join(exts_htmltype.keys())
        try:
            ext = exts_htmltype[ext]
        except:
            raise ValueError('This video extension cannot be displayed in the IPython Notebook. Allowed extensions: ' + allowed_exts)
    if filetype in ['audio', 'video']:
        duration = ffmpeg_parse_infos(filename)['duration']
        if duration > maxduration:
            raise ValueError("The duration of video %s (%.1f) exceeds the 'maxduration' " % (filename, duration) + "attribute. You can increase 'maxduration', by passing 'maxduration' parameterto ipython_display function.But note that embedding large videos may take all the memory away !")
    with open(filename, 'rb') as f:
        data = b64encode(f.read()).decode('utf-8')
    template = templates[filetype]
    result = template % {'data': data, 'options': options, 'ext': ext}
    if center:
        result = '<div align=middle>%s</div>' % result
    return result