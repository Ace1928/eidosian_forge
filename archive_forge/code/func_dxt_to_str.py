from struct import pack, unpack, calcsize
def dxt_to_str(dxt):
    if dxt == DDS_DXT1:
        return 's3tc_dxt1'
    elif dxt == DDS_DXT2:
        return 's3tc_dxt2'
    elif dxt == DDS_DXT3:
        return 's3tc_dxt3'
    elif dxt == DDS_DXT4:
        return 's3tc_dxt4'
    elif dxt == DDS_DXT5:
        return 's3tc_dxt5'
    elif dxt == 0:
        return 'rgba'
    elif dxt == 1:
        return 'alpha'
    elif dxt == 2:
        return 'luminance'
    elif dxt == 3:
        return 'luminance_alpha'