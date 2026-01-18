from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
def fitLine(self, program, totalLength):
    """fit words (and other things) onto a line"""
    from reportlab.pdfbase.pdfmetrics import stringWidth
    usedIndent = self.indent
    maxLength = totalLength - usedIndent - self.rightIndent
    done = 0
    line = []
    cursor = 0
    lineIsFull = 0
    currentLength = 0
    maxcursor = len(program)
    needspace = 0
    first = 1
    terminated = None
    fontName = self.fontName
    fontSize = self.fontSize
    spacewidth = stringWidth(' ', fontName, fontSize)
    justStrings = 1
    while not done and cursor < maxcursor:
        opcode = program[cursor]
        if isinstance(opcode, str) or hasattr(opcode, 'width'):
            lastneedspace = needspace
            needspace = 0
            if hasattr(opcode, 'width'):
                justStrings = 0
                width = opcode.width(self)
                needspace = 0
            else:
                saveopcode = opcode
                opcode = opcode.strip()
                if opcode:
                    width = stringWidth(opcode, fontName, fontSize)
                else:
                    width = 0
                if saveopcode and (width or currentLength):
                    needspace = saveopcode[-1] == ' '
                else:
                    needspace = 0
            fullwidth = width
            if lastneedspace:
                fullwidth = width + spacewidth
            newlength = currentLength + fullwidth
            if newlength > maxLength and (not first):
                done = 1
                lineIsFull = 1
            else:
                if lastneedspace:
                    line.append(spacewidth)
                if opcode:
                    line.append(opcode)
                if abs(width) > TOOSMALLSPACE:
                    line.append(-width)
                    currentLength = newlength
            first = 0
        elif isinstance(opcode, float):
            justStrings = 0
            aopcode = abs(opcode)
            if aopcode > TOOSMALLSPACE:
                nextLength = currentLength + aopcode
                if nextLength > maxLength and (not first):
                    done = 1
                elif aopcode > TOOSMALLSPACE:
                    currentLength = nextLength
                    line.append(opcode)
                first = 0
        elif isinstance(opcode, tuple):
            justStrings = 0
            indicator = opcode[0]
            if indicator == 'nextLine':
                line.append(opcode)
                cursor += 1
                terminated = done = 1
            elif indicator == 'color':
                oldcolor = self.fontColor
                i, colorname = opcode
                if isinstance(colorname, str):
                    color = self.fontColor = getattr(colors, colorname)
                else:
                    color = self.fontColor = colorname
                line.append(opcode)
            elif indicator == 'face':
                i, fontname = opcode
                fontName = self.fontName = fontname
                spacewidth = stringWidth(' ', fontName, fontSize)
                line.append(opcode)
            elif indicator == 'size':
                i, fontsize = opcode
                size = abs(float(fontsize))
                if isinstance(fontsize, str):
                    if fontsize[:1] == '+':
                        fontSize = self.fontSize = self.fontSize + size
                    elif fontsize[:1] == '-':
                        fontSize = self.fontSize = self.fontSize - size
                    else:
                        fontSize = self.fontSize = size
                else:
                    fontSize = self.fontSize = size
                spacewidth = stringWidth(' ', fontName, fontSize)
                line.append(opcode)
            elif indicator == 'leading':
                i, leading = opcode
                self.leading = leading
                line.append(opcode)
            elif indicator == 'indent':
                i, increment = opcode
                indent = self.indent = self.indent + increment
                if first:
                    usedIndent = max(indent, usedIndent)
                    maxLength = totalLength - usedIndent - self.rightIndent
                line.append(opcode)
            elif indicator == 'push':
                self.pushTextState()
                line.append(opcode)
            elif indicator == 'pop':
                try:
                    self.popTextState()
                except:
                    raise
                fontName = self.fontName
                fontSize = self.fontSize
                spacewidth = stringWidth(' ', fontName, fontSize)
                line.append(opcode)
            elif indicator == 'bullet':
                i, bullet, indent, font, size = opcode
                indent = indent + self.baseindent
                opcode = (i, bullet, indent, font, size)
                if not first:
                    raise ValueError('bullet not at beginning of line')
                bulletwidth = float(stringWidth(bullet, font, size))
                spacewidth = float(stringWidth(' ', font, size))
                bulletmin = indent + spacewidth + bulletwidth
                usedIndent = max(bulletmin, usedIndent)
                if first:
                    maxLength = totalLength - usedIndent - self.rightIndent
                line.append(opcode)
            elif indicator == 'rightIndent':
                i, increment = opcode
                self.rightIndent = self.rightIndent + increment
                if first:
                    maxLength = totalLength - usedIndent - self.rightIndent
                line.append(opcode)
            elif indicator == 'rise':
                i, rise = opcode
                newrise = self.rise = self.rise + rise
                line.append(opcode)
            elif indicator == 'align':
                i, alignment = opcode
                self.alignment = alignment
                line.append(opcode)
            elif indicator == 'lineOperation':
                i, handler = opcode
                line.append(opcode)
                self.lineOpHandlers = self.lineOpHandlers + [handler]
            elif indicator == 'endLineOperation':
                i, handler = opcode
                h = self.lineOpHandlers[:]
                h.remove(handler)
                self.lineOpHandlers = h
                line.append(opcode)
            else:
                raise ValueError("at format time don't understand indicator " + repr(indicator))
        else:
            raise ValueError('op must be string, float, instance, or tuple ' + repr(opcode))
        if not done:
            cursor += 1
    if not terminated:
        line.append(('nextLine', 0))
    return (lineIsFull, line, cursor, currentLength, usedIndent, maxLength, justStrings)