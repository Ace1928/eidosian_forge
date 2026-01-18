def decodestring(s, header=False):
    if a2b_qp is not None:
        return a2b_qp(s, header=header)
    from io import BytesIO
    infp = BytesIO(s)
    outfp = BytesIO()
    decode(infp, outfp, header=header)
    return outfp.getvalue()