def parse_mt_only(s, start):
    mt, k = parse_media_type(s, start, with_parameters=False)
    ct = content_type()
    ct.major = mt[0]
    ct.minor = mt[1]
    return (ct, k)