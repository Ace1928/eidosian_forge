from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
def runOpCodes(self, program, canvas, textobject):
    """render the line(s)"""
    escape = canvas._escape
    code = textobject._code
    startstate = self.__dict__.copy()
    font = None
    size = None
    textobject.setFillColor(self.fontColor)
    xstart = self.x
    thislineindent = self.indent
    thislinerightIndent = self.rightIndent
    indented = 0
    for opcode in program:
        if isinstance(opcode, str) or hasattr(opcode, 'width'):
            if not indented:
                if abs(thislineindent) > TOOSMALLSPACE:
                    code.append('%s Td' % fp_str(thislineindent, 0))
                    self.x += thislineindent
                for handler in self.lineOpHandlers:
                    handler.start_at(self.x, self.y, self, canvas, textobject)
            indented = 1
            if font != self.fontName or size != self.fontSize:
                font = self.fontName
                size = self.fontSize
                textobject.setFont(font, size)
            if isinstance(opcode, str):
                textobject.textOut(opcode)
            else:
                opcode.execute(self, textobject, canvas)
        elif isinstance(opcode, float):
            opcode = abs(opcode)
            if opcode > TOOSMALLSPACE:
                code.append('%s Td' % fp_str(opcode, 0))
                self.x += opcode
        elif isinstance(opcode, tuple):
            indicator = opcode[0]
            if indicator == 'nextLine':
                i, endallmarks = opcode
                x = self.x
                y = self.y
                newy = self.y = self.y - self.leading
                newx = self.x = xstart
                thislineindent = self.indent
                thislinerightIndent = self.rightIndent
                indented = 0
                for handler in self.lineOpHandlers:
                    handler.end_at(x, y, self, canvas, textobject)
                textobject.setTextOrigin(newx, newy)
            elif indicator == 'color':
                oldcolor = self.fontColor
                i, colorname = opcode
                if isinstance(colorname, str):
                    color = self.fontColor = getattr(colors, colorname)
                else:
                    color = self.fontColor = colorname
                if color != oldcolor:
                    textobject.setFillColor(color)
            elif indicator == 'face':
                i, fontname = opcode
                self.fontName = fontname
            elif indicator == 'size':
                i, fontsize = opcode
                size = abs(float(fontsize))
                if isinstance(fontsize, str):
                    if fontsize[:1] == '+':
                        self.fontSize += size
                    elif fontsize[:1] == '-':
                        self.fontSize -= size
                    else:
                        self.fontSize = size
                else:
                    self.fontSize = size
                fontSize = self.fontSize
                textobject.setFont(self.fontName, self.fontSize)
            elif indicator == 'leading':
                i, leading = opcode
                self.leading = leading
            elif indicator == 'indent':
                i, increment = opcode
                indent = self.indent = self.indent + increment
                thislineindent = max(thislineindent, indent)
            elif indicator == 'push':
                self.pushTextState()
            elif indicator == 'pop':
                oldcolor = self.fontColor
                oldfont = self.fontName
                oldsize = self.fontSize
                self.popTextState()
                if oldcolor != self.fontColor:
                    textobject.setFillColor(self.fontColor)
            elif indicator == 'wordSpacing':
                i, ws = opcode
                textobject.setWordSpace(ws)
            elif indicator == 'bullet':
                i, bullet, indent, font, size = opcode
                if abs(self.x - xstart) > TOOSMALLSPACE:
                    raise ValueError('bullet not at beginning of line')
                bulletwidth = float(stringWidth(bullet, font, size))
                spacewidth = float(stringWidth(' ', font, size))
                bulletmin = indent + spacewidth + bulletwidth
                if bulletmin > thislineindent:
                    thislineindent = bulletmin
                textobject.moveCursor(indent, 0)
                textobject.setFont(font, size)
                textobject.textOut(bullet)
                textobject.moveCursor(-indent, 0)
                textobject.setFont(self.fontName, self.fontSize)
            elif indicator == 'rightIndent':
                i, increment = opcode
                self.rightIndent = self.rightIndent + increment
            elif indicator == 'rise':
                i, rise = opcode
                newrise = self.rise = self.rise + rise
                textobject.setRise(newrise)
            elif indicator == 'align':
                i, alignment = opcode
                self.alignment = alignment
            elif indicator == 'lineOperation':
                i, handler = opcode
                handler.start_at(self.x, self.y, self, canvas, textobject)
                self.lineOpHandlers = self.lineOpHandlers + [handler]
            elif indicator == 'endLineOperation':
                i, handler = opcode
                handler.end_at(self.x, self.y, self, canvas, textobject)
                newh = self.lineOpHandlers = self.lineOpHandlers[:]
                if handler in newh:
                    self.lineOpHandlers.remove(handler)
                else:
                    pass
            else:
                raise ValueError("don't understand indicator " + repr(indicator))
        else:
            raise ValueError('op must be string float or tuple ' + repr(opcode))
    laststate = self.__dict__.copy()
    self.__dict__.update(startstate)
    return laststate