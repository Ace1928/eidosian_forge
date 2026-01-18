import argparse
import bisect
import os
import sys
import statistics
from typing import Dict, List, Set
from . import fitz
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