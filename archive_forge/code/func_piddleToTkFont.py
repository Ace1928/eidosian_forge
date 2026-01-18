import tkFont
import Tkinter
import rdkit.sping.pid
def piddleToTkFont(self, font):
    """Return a tkFont instance based on the pid-style FONT"""
    if font is None:
        return ''
    size = 12
    family = 'Times'
    weight = 'normal'
    slant = 'roman'
    underline = 'false'
    if font.face:
        f = font.face.lower()
        if f in self.__alt_faces:
            family = self.__alt_faces[f]
        else:
            family = font.face
    size = font.size or 12
    if font.bold:
        weight = 'bold'
    if font.italic:
        slant = 'italic'
    if font.underline:
        underline = 'true'
    key = (family, size, weight, slant, underline)
    if key in self.font_cache:
        font = self.font_cache[key]
    else:
        font = tkFont.Font(self.master, family=family, size=size, weight=weight, slant=slant, underline=underline)
        self.font_cache[family, size, weight, slant, underline] = font
    return font