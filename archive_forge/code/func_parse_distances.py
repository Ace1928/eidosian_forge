import re
def parse_distances(lines, results):
    """Parse amino acid sequence distance results."""
    distances = {}
    sequences = []
    raw_aa_distances_flag = False
    ml_aa_distances_flag = False
    matrix_row_re = re.compile('(.+)\\s{5,15}')
    for line in lines:
        line_floats_res = line_floats_re.findall(line)
        line_floats = [float(val) for val in line_floats_res]
        if 'AA distances' in line:
            raw_aa_distances_flag = True
            ml_aa_distances_flag = False
        elif 'ML distances of aa seqs.' in line:
            ml_aa_distances_flag = True
            raw_aa_distances_flag = False
        matrix_row_res = matrix_row_re.match(line)
        if matrix_row_res and (raw_aa_distances_flag or ml_aa_distances_flag):
            seq_name = matrix_row_res.group(1).strip()
            if seq_name not in sequences:
                sequences.append(seq_name)
            if raw_aa_distances_flag:
                if distances.get('raw') is None:
                    distances['raw'] = {}
                distances['raw'][seq_name] = {}
                for i in range(len(line_floats)):
                    distances['raw'][seq_name][sequences[i]] = line_floats[i]
                    distances['raw'][sequences[i]][seq_name] = line_floats[i]
            else:
                if distances.get('ml') is None:
                    distances['ml'] = {}
                distances['ml'][seq_name] = {}
                for i in range(len(line_floats)):
                    distances['ml'][seq_name][sequences[i]] = line_floats[i]
                    distances['ml'][sequences[i]][seq_name] = line_floats[i]
    if distances:
        results['distances'] = distances
    return results