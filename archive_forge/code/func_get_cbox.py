import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def get_cbox(self, bbox_mode):
    """
        Return an outline's 'control box'. The control box encloses all the
        outline's points, including Bezier control points. Though it coincides
        with the exact bounding box for most glyphs, it can be slightly larger
        in some situations (like when rotating an outline which contains Bezier
        outside arcs).

        Computing the control box is very fast, while getting the bounding box
        can take much more time as it needs to walk over all segments and arcs
        in the outline. To get the latter, you can use the 'ftbbox' component
        which is dedicated to this single task.

        :param mode: The mode which indicates how to interpret the returned
                     bounding box values.

        **Note**:

          Coordinates are relative to the glyph origin, using the y upwards
          convention.

          If the glyph has been loaded with FT_LOAD_NO_SCALE, 'bbox_mode' must be
          set to FT_GLYPH_BBOX_UNSCALED to get unscaled font units in 26.6 pixel
          format. The value FT_GLYPH_BBOX_SUBPIXELS is another name for this
          constant.

          Note that the maximum coordinates are exclusive, which means that one
          can compute the width and height of the glyph image (be it in integer
          or 26.6 pixels) as:

          width  = bbox.xMax - bbox.xMin;
          height = bbox.yMax - bbox.yMin;

          Note also that for 26.6 coordinates, if 'bbox_mode' is set to
          FT_GLYPH_BBOX_GRIDFIT, the coordinates will also be grid-fitted, which
          corresponds to:

          bbox.xMin = FLOOR(bbox.xMin);
          bbox.yMin = FLOOR(bbox.yMin);
          bbox.xMax = CEILING(bbox.xMax);
          bbox.yMax = CEILING(bbox.yMax);

          To get the bbox in pixel coordinates, set 'bbox_mode' to
          FT_GLYPH_BBOX_TRUNCATE.

          To get the bbox in grid-fitted pixel coordinates, set 'bbox_mode' to
          FT_GLYPH_BBOX_PIXELS.
        """
    bbox = FT_BBox()
    FT_Glyph_Get_CBox(byref(self._FT_Glyph.contents), bbox_mode, byref(bbox))
    return BBox(bbox)