import re
def parse_model(lines, results):
    """Parse an individual NSsites model's results."""
    parameters = {}
    SEs_flag = False
    dS_tree_flag = False
    dN_tree_flag = False
    w_tree_flag = False
    num_params = None
    tree_re = re.compile("^\\([\\w #:',.()]*\\);\\s*$")
    branch_re = re.compile('\\s+(\\d+\\.\\.\\d+)[\\s+\\d+\\.\\d+]+')
    model_params_re = re.compile('(?<!\\S)([a-z]\\d?)\\s*=\\s+(\\d+\\.\\d+)')
    for line in lines:
        line_floats_res = line_floats_re.findall(line)
        line_floats = [float(val) for val in line_floats_res]
        branch_res = branch_re.match(line)
        model_params = model_params_re.findall(line)
        if 'lnL(ntime:' in line and line_floats:
            results['lnL'] = line_floats[0]
            np_res = re.match('lnL\\(ntime:\\s+\\d+\\s+np:\\s+(\\d+)\\)', line)
            if np_res is not None:
                num_params = int(np_res.group(1))
        elif len(line_floats) == num_params and (not SEs_flag):
            parameters['parameter list'] = line.strip()
        elif 'SEs for parameters:' in line:
            SEs_flag = True
        elif SEs_flag and len(line_floats) == num_params:
            parameters['SEs'] = line.strip()
            SEs_flag = False
        elif 'tree length =' in line and line_floats:
            results['tree length'] = line_floats[0]
        elif tree_re.match(line) is not None:
            if ':' in line or '#' in line:
                if dS_tree_flag:
                    results['dS tree'] = line.strip()
                    dS_tree_flag = False
                elif dN_tree_flag:
                    results['dN tree'] = line.strip()
                    dN_tree_flag = False
                elif w_tree_flag:
                    results['omega tree'] = line.strip()
                    w_tree_flag = False
                else:
                    results['tree'] = line.strip()
        elif 'dS tree:' in line:
            dS_tree_flag = True
        elif 'dN tree:' in line:
            dN_tree_flag = True
        elif 'w ratios as labels for TreeView:' in line:
            w_tree_flag = True
        elif 'rates for' in line and line_floats:
            line_floats.insert(0, 1.0)
            parameters['rates'] = line_floats
        elif 'kappa (ts/tv)' in line and line_floats:
            parameters['kappa'] = line_floats[0]
        elif 'omega (dN/dS)' in line and line_floats:
            parameters['omega'] = line_floats[0]
        elif 'w (dN/dS)' in line and line_floats:
            parameters['omega'] = line_floats
        elif 'gene # ' in line:
            gene_num = int(re.match('gene # (\\d+)', line).group(1))
            if parameters.get('genes') is None:
                parameters['genes'] = {}
            parameters['genes'][gene_num] = {'kappa': line_floats[0], 'omega': line_floats[1]}
        elif 'tree length for dN' in line and line_floats:
            parameters['dN'] = line_floats[0]
        elif 'tree length for dS' in line and line_floats:
            parameters['dS'] = line_floats[0]
        elif line[0:2] == 'p:' or line[0:10] == 'proportion':
            site_classes = parse_siteclass_proportions(line_floats)
            parameters['site classes'] = site_classes
        elif line[0:2] == 'w:':
            site_classes = parameters.get('site classes')
            site_classes = parse_siteclass_omegas(line, site_classes)
            parameters['site classes'] = site_classes
        elif 'branch type ' in line:
            branch_type = re.match('branch type (\\d)', line)
            if branch_type:
                site_classes = parameters.get('site classes')
                branch_type_no = int(branch_type.group(1))
                site_classes = parse_clademodelc(branch_type_no, line_floats, site_classes)
                parameters['site classes'] = site_classes
        elif line[0:12] == 'foreground w':
            site_classes = parameters.get('site classes')
            site_classes = parse_branch_site_a(True, line_floats, site_classes)
            parameters['site classes'] = site_classes
        elif line[0:12] == 'background w':
            site_classes = parameters.get('site classes')
            site_classes = parse_branch_site_a(False, line_floats, site_classes)
            parameters['site classes'] = site_classes
        elif branch_res is not None and line_floats:
            branch = branch_res.group(1)
            if parameters.get('branches') is None:
                parameters['branches'] = {}
            params = line.strip().split()[1:]
            parameters['branches'][branch] = {'t': float(params[0].strip()), 'N': float(params[1].strip()), 'S': float(params[2].strip()), 'omega': float(params[3].strip()), 'dN': float(params[4].strip()), 'dS': float(params[5].strip()), 'N*dN': float(params[6].strip()), 'S*dS': float(params[7].strip())}
        elif model_params:
            float_model_params = []
            for param in model_params:
                float_model_params.append((param[0], float(param[1])))
            parameters.update(dict(float_model_params))
    if parameters:
        results['parameters'] = parameters
    return results