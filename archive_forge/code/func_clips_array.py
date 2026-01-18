import numpy as np
from moviepy.audio.AudioClip import CompositeAudioClip
from moviepy.video.VideoClip import ColorClip, VideoClip
def clips_array(array, rows_widths=None, cols_widths=None, bg_color=None):
    """

    rows_widths
      widths of the different rows in pixels. If None, is set automatically.

    cols_widths
      widths of the different colums in pixels. If None, is set automatically.

    cols_widths
    
    bg_color
       Fill color for the masked and unfilled regions. Set to None for these
       regions to be transparent (will be slower).

    """
    array = np.array(array)
    sizes_array = np.array([[c.size for c in line] for line in array])
    if rows_widths is None:
        rows_widths = sizes_array[:, :, 1].max(axis=1)
    if cols_widths is None:
        cols_widths = sizes_array[:, :, 0].max(axis=0)
    xx = np.cumsum([0] + list(cols_widths))
    yy = np.cumsum([0] + list(rows_widths))
    for j, (x, cw) in enumerate(zip(xx[:-1], cols_widths)):
        for i, (y, rw) in enumerate(zip(yy[:-1], rows_widths)):
            clip = array[i, j]
            w, h = clip.size
            if w < cw or h < rw:
                clip = CompositeVideoClip([clip.set_position('center')], size=(cw, rw), bg_color=bg_color).set_duration(clip.duration)
            array[i, j] = clip.set_position((x, y))
    return CompositeVideoClip(array.flatten(), size=(xx[-1], yy[-1]), bg_color=bg_color)