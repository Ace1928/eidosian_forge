import os
def calc_mode_diff(cur_mode, mode, keep_exe=True, not_set='not set: ', set='set: '):
    """
    Gives the difference between the actual mode of the file and the
    given mode.  If ``keep_exe`` is true, then if the mode doesn't
    include any executable information the executable information will
    simply be ignored.  High bits are also always ignored (except
    suid/sgid and sticky bit).

    Returns a list of differences (empty list if no differences)
    """
    for exe_bit in exe_bits:
        if mode & exe_bit:
            keep_exe = False
    diffs = []
    isdir = os.path.isdir(filename)
    for bit, file_desc, dir_desc in modes:
        if keep_exe and bit in exe_bits:
            continue
        if isdir:
            desc = dir_desc
        else:
            desc = file_desc
        if mode & bit and (not cur_mode & bit):
            diffs.append(not_set + desc)
        if not mode & bit and cur_mode & bit:
            diffs.append(set + desc)
    return diffs