from __future__ import print_function, unicode_literals
import six
def get_font_version(font_face):
    from freetype import TT_NAME_ID_VERSION_STRING
    for i in range(font_face.sfnt_name_count):
        name = font_face.get_sfnt_name(i)
        if name.name_id == TT_NAME_ID_VERSION_STRING:
            return name.string
    return 'unknown'