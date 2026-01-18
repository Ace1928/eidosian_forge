import io
import math
import os
import typing
import weakref
def get_ocmd(doc: fitz.Document, xref: int) -> dict:
    """Return the definition of an OCMD (optional content membership dictionary).

    Recognizes PDF dict keys /OCGs (PDF array of OCGs), /P (policy string) and
    /VE (visibility expression, PDF array). Via string manipulation, this
    info is converted to a Python dictionary with keys "xref", "ocgs", "policy"
    and "ve" - ready to recycle as input for 'set_ocmd()'.
    """
    if xref not in range(doc.xref_length()):
        raise ValueError('bad xref')
    text = doc.xref_object(xref, compressed=True)
    if '/Type/OCMD' not in text:
        raise ValueError('bad object type')
    textlen = len(text)
    p0 = text.find('/OCGs[')
    p1 = text.find(']', p0)
    if p0 < 0 or p1 < 0:
        ocgs = None
    else:
        ocgs = text[p0 + 6:p1].replace('0 R', ' ').split()
        ocgs = list(map(int, ocgs))
    p0 = text.find('/P/')
    if p0 < 0:
        policy = None
    else:
        p1 = text.find('ff', p0)
        if p1 < 0:
            p1 = text.find('on', p0)
        if p1 < 0:
            raise ValueError('bad object at xref')
        else:
            policy = text[p0 + 3:p1 + 2]
    p0 = text.find('/VE[')
    if p0 < 0:
        ve = None
    else:
        lp = rp = 0
        p1 = p0
        while lp < 1 or lp != rp:
            p1 += 1
            if not p1 < textlen:
                raise ValueError('bad object at xref')
            if text[p1] == '[':
                lp += 1
            if text[p1] == ']':
                rp += 1
        ve = text[p0 + 3:p1 + 1]
        ve = ve.replace('/And', '"and",').replace('/Not', '"not",').replace('/Or', '"or",')
        ve = ve.replace(' 0 R]', ']').replace(' 0 R', ',').replace('][', '],[')
        import json
        try:
            ve = json.loads(ve)
        except Exception:
            fitz.exception_info()
            fitz.message(f'bad /VE key: {ve!r}')
            raise
    return {'xref': xref, 'ocgs': ocgs, 'policy': policy, 've': ve}