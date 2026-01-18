import re
def parse_freqs(lines, parameters):
    """Parse the basepair frequencies."""
    root_re = re.compile('Note: node (\\d+) is root.')
    branch_freqs_found = False
    base_freqs_found = False
    for line in lines:
        line_floats_res = line_floats_re.findall(line)
        line_floats = [float(val) for val in line_floats_res]
        if 'Base frequencies' in line and line_floats:
            base_frequencies = {}
            base_frequencies['T'] = line_floats[0]
            base_frequencies['C'] = line_floats[1]
            base_frequencies['A'] = line_floats[2]
            base_frequencies['G'] = line_floats[3]
            parameters['base frequencies'] = base_frequencies
        elif 'base frequency parameters' in line:
            base_freqs_found = True
        elif 'Base frequencies' in line and (not line_floats):
            base_freqs_found = True
        elif base_freqs_found and line_floats:
            base_frequencies = {}
            base_frequencies['T'] = line_floats[0]
            base_frequencies['C'] = line_floats[1]
            base_frequencies['A'] = line_floats[2]
            base_frequencies['G'] = line_floats[3]
            parameters['base frequencies'] = base_frequencies
            base_freqs_found = False
        elif 'freq: ' in line and line_floats:
            parameters['rate frequencies'] = line_floats
        elif '(frequency parameters for branches)' in line:
            parameters['nodes'] = {}
            branch_freqs_found = True
        elif branch_freqs_found:
            if line_floats:
                node_res = re.match('Node \\#(\\d+)', line)
                node_num = int(node_res.group(1))
                node = {'root': False}
                node['frequency parameters'] = line_floats[:4]
                if len(line_floats) > 4:
                    node['base frequencies'] = {'T': line_floats[4], 'C': line_floats[5], 'A': line_floats[6], 'G': line_floats[7]}
                parameters['nodes'][node_num] = node
            else:
                root_res = root_re.match(line)
                if root_res is not None:
                    root_node = int(root_res.group(1))
                    parameters['nodes'][root_node]['root'] = True
                    branch_freqs_found = False
    return parameters