import reportlab
def asciiBase85Encode(input):
    """Encodes input using ASCII-Base85 coding.

        This is a compact encoding used for binary data within
        a PDF file.  Four bytes of binary data become five bytes of
        ASCII.  This is the default method used for encoding images."""
    doOrd = isUnicode(input)
    whole_word_count, remainder_size = divmod(len(input), 4)
    cut = 4 * whole_word_count
    body, lastbit = (input[0:cut], input[cut:])
    out = [].append
    for i in range(whole_word_count):
        offset = i * 4
        b1 = body[offset]
        b2 = body[offset + 1]
        b3 = body[offset + 2]
        b4 = body[offset + 3]
        if doOrd:
            b1 = ord(b1)
            b2 = ord(b2)
            b3 = ord(b3)
            b4 = ord(b4)
        if b1 < 128:
            num = ((b1 << 8 | b2) << 8 | b3) << 8 | b4
        else:
            num = 16777216 * b1 + 65536 * b2 + 256 * b3 + b4
        if num == 0:
            out('z')
        else:
            temp, c5 = divmod(num, 85)
            temp, c4 = divmod(temp, 85)
            temp, c3 = divmod(temp, 85)
            c1, c2 = divmod(temp, 85)
            assert 85 ** 4 * c1 + 85 ** 3 * c2 + 85 ** 2 * c3 + 85 * c4 + c5 == num, 'dodgy code!'
            out(chr(c1 + 33))
            out(chr(c2 + 33))
            out(chr(c3 + 33))
            out(chr(c4 + 33))
            out(chr(c5 + 33))
    if remainder_size > 0:
        lastbit += (4 - len(lastbit)) * ('\x00' if doOrd else b'\x00')
        b1 = lastbit[0]
        b2 = lastbit[1]
        b3 = lastbit[2]
        b4 = lastbit[3]
        if doOrd:
            b1 = ord(b1)
            b2 = ord(b2)
            b3 = ord(b3)
            b4 = ord(b4)
        num = 16777216 * b1 + 65536 * b2 + 256 * b3 + b4
        temp, c5 = divmod(num, 85)
        temp, c4 = divmod(temp, 85)
        temp, c3 = divmod(temp, 85)
        c1, c2 = divmod(temp, 85)
        lastword = chr(c1 + 33) + chr(c2 + 33) + chr(c3 + 33) + chr(c4 + 33) + chr(c5 + 33)
        out(lastword[0:remainder_size + 1])
    out('~>')
    return ''.join(out.__self__)