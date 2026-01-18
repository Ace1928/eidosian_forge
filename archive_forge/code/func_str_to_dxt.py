from struct import pack, unpack, calcsize
def str_to_dxt(dxt):
    if dxt == 's3tc_dxt1':
        return DDS_DXT1
    if dxt == 's3tc_dxt2':
        return DDS_DXT2
    if dxt == 's3tc_dxt3':
        return DDS_DXT3
    if dxt == 's3tc_dxt4':
        return DDS_DXT4
    if dxt == 's3tc_dxt5':
        return DDS_DXT5
    if dxt == 'rgba':
        return 0
    if dxt == 'alpha':
        return 1
    if dxt == 'luminance':
        return 2
    if dxt == 'luminance_alpha':
        return 3