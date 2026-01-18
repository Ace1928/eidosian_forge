import re
def parse_yn00(lines, results, sequences):
    """Parse the Yang & Nielsen (2000) part of the results.

    Yang & Nielsen results are organized in a table with
    each row comprising one pairwise species comparison.
    Rows are labeled by sequence number rather than by
    sequence name.
    """
    for line in lines:
        line_floats_res = re.findall('-*\\d+\\.\\d+', line)
        line_floats = [float(val) for val in line_floats_res]
        row_res = re.match('\\s+(\\d+)\\s+(\\d+)', line)
        if row_res is not None:
            seq1 = int(row_res.group(1))
            seq2 = int(row_res.group(2))
            seq_name1 = sequences[seq1 - 1]
            seq_name2 = sequences[seq2 - 1]
            YN00 = {}
            YN00['S'] = line_floats[0]
            YN00['N'] = line_floats[1]
            YN00['t'] = line_floats[2]
            YN00['kappa'] = line_floats[3]
            YN00['omega'] = line_floats[4]
            YN00['dN'] = line_floats[5]
            YN00['dN SE'] = line_floats[6]
            YN00['dS'] = line_floats[7]
            YN00['dS SE'] = line_floats[8]
            results[seq_name1][seq_name2]['YN00'] = YN00
            results[seq_name2][seq_name1]['YN00'] = YN00
            seq_name1 = None
            seq_name2 = None
    return results