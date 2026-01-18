import argparse
import bisect
import os
import sys
import statistics
from typing import Dict, List, Set
from . import fitz
def page_layout(page, textout, GRID, fontsize, noformfeed, skip_empty, flags):
    eop = b'\n' if noformfeed else bytes([12])

    def find_line_index(values: List[int], value: int) -> int:
        """Find the right row coordinate.

        Args:
            values: (list) y-coordinates of rows.
            value: (int) lookup for this value (y-origin of char).
        Returns:
            y-ccordinate of appropriate line for value.
        """
        i = bisect.bisect_right(values, value)
        if i:
            return values[i - 1]
        raise RuntimeError('Line for %g not found in %s' % (value, values))

    def curate_rows(rows: Set[int], GRID) -> List:
        rows = list(rows)
        rows.sort()
        nrows = [rows[0]]
        for h in rows[1:]:
            if h >= nrows[-1] + GRID:
                nrows.append(h)
        return nrows

    def process_blocks(blocks: List[Dict], page: fitz.Page):
        rows = set()
        page_width = page.rect.width
        page_height = page.rect.height
        rowheight = page_height
        left = page_width
        right = 0
        chars = []
        for block in blocks:
            for line in block['lines']:
                if line['dir'] != (1, 0):
                    continue
                x0, y0, x1, y1 = line['bbox']
                if y1 < 0 or y0 > page.rect.height:
                    continue
                height = y1 - y0
                if rowheight > height:
                    rowheight = height
                for span in line['spans']:
                    if span['size'] <= fontsize:
                        continue
                    for c in span['chars']:
                        x0, _, x1, _ = c['bbox']
                        cwidth = x1 - x0
                        ox, oy = c['origin']
                        oy = int(round(oy))
                        rows.add(oy)
                        ch = c['c']
                        if left > ox and ch != ' ':
                            left = ox
                        if right < x1:
                            right = x1
                        if cwidth == 0 and chars != []:
                            old_ch, old_ox, old_oy, old_cwidth = chars[-1]
                            if old_oy == oy:
                                if old_ch != chr(64256):
                                    lig = joinligature(old_ch + ch)
                                elif ch == 'i':
                                    lig = chr(64259)
                                elif ch == 'l':
                                    lig = chr(64260)
                                else:
                                    lig = old_ch
                                chars[-1] = (lig, old_ox, old_oy, old_cwidth)
                                continue
                        chars.append((ch, ox, oy, cwidth))
        return (chars, rows, left, right, rowheight)

    def joinligature(lig: str) -> str:
        """Return ligature character for a given pair / triple of characters.

        Args:
            lig: (str) 2/3 characters, e.g. "ff"
        Returns:
            Ligature, e.g. "ff" -> chr(0xFB00)
        """
        if lig == 'ff':
            return chr(64256)
        elif lig == 'fi':
            return chr(64257)
        elif lig == 'fl':
            return chr(64258)
        elif lig == 'ffi':
            return chr(64259)
        elif lig == 'ffl':
            return chr(64260)
        elif lig == 'ft':
            return chr(64261)
        elif lig == 'st':
            return chr(64262)
        return lig

    def make_textline(left, slot, minslot, lchars):
        """Produce the text of one output line.

        Args:
            left: (float) left most coordinate used on page
            slot: (float) avg width of one character in any font in use.
            minslot: (float) min width for the characters in this line.
            chars: (list[tuple]) characters of this line.
        Returns:
            text: (str) text string for this line
        """
        text = ''
        old_char = ''
        old_x1 = 0
        old_ox = 0
        if minslot <= fitz.EPSILON:
            raise RuntimeError('program error: minslot too small = %g' % minslot)
        for c in lchars:
            char, ox, _, cwidth = c
            ox = ox - left
            x1 = ox + cwidth
            if old_char == char and ox - old_ox <= cwidth * 0.2:
                continue
            if char == ' ' and (old_x1 - ox) / cwidth > 0.8:
                continue
            old_char = char
            if ox < old_x1 + minslot:
                text += char
                old_x1 = x1
                old_ox = ox
                continue
            if char == ' ':
                continue
            delta = int(ox / slot) - len(text)
            if ox > old_x1 and delta > 1:
                text += ' ' * delta
            text += char
            old_x1 = x1
            old_ox = ox
        return text.rstrip()
    blocks = page.get_text('rawdict', flags=flags)['blocks']
    chars, rows, left, right, rowheight = process_blocks(blocks, page)
    if chars == []:
        if not skip_empty:
            textout.write(eop)
        return
    rows = curate_rows(rows, GRID)
    chars.sort(key=lambda c: c[1])
    lines = {}
    for c in chars:
        _, _, oy, _ = c
        y = find_line_index(rows, oy)
        lchars = lines.get(y, [])
        lchars.append(c)
        lines[y] = lchars
    keys = list(lines.keys())
    keys.sort()
    slot = right - left
    minslots = {}
    for k in keys:
        lchars = lines[k]
        ccount = len(lchars)
        if ccount < 2:
            minslots[k] = 1
            continue
        widths = [c[3] for c in lchars]
        widths.sort()
        this_slot = statistics.median(widths)
        if this_slot < slot:
            slot = this_slot
        minslots[k] = widths[0]
    rowheight = rowheight * (rows[-1] - rows[0]) / (rowheight * len(rows)) * 1.2
    rowpos = rows[0]
    textout.write(b'\n')
    for k in keys:
        while rowpos < k:
            textout.write(b'\n')
            rowpos += rowheight
        text = make_textline(left, slot, minslots[k], lines[k])
        textout.write((text + '\n').encode('utf8', errors='surrogatepass'))
        rowpos = k + rowheight
    textout.write(eop)