from struct import pack, unpack, calcsize
def dxt_size(w, h, dxt):
    w = max(1, w // 4)
    h = max(1, h // 4)
    if dxt == DDS_DXT1:
        return w * h * 8
    elif dxt in (DDS_DXT2, DDS_DXT3, DDS_DXT4, DDS_DXT5):
        return w * h * 16
    return -1