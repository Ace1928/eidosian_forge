import re
def parse_nssites(lines, results, multi_models, multi_genes):
    """Determine which NSsites models are present and parse them."""
    ns_sites = {}
    model_re = re.compile('Model (\\d+):\\s+(.+)')
    gene_re = re.compile('Gene\\s+([0-9]+)\\s+.+')
    siteclass_model = results.get('site-class model')
    if not multi_models:
        if siteclass_model is None:
            siteclass_model = 'one-ratio'
        current_model = {'one-ratio': 0, 'NearlyNeutral': 1, 'PositiveSelection': 2, 'discrete': 3, 'beta': 7, 'beta&w>1': 8, 'M2a_rel': 22}[siteclass_model]
        if multi_genes:
            genes = results['genes']
            current_gene = None
            gene_start = None
            model_results = None
            for line_num, line in enumerate(lines):
                gene_res = gene_re.match(line)
                if gene_res:
                    if current_gene is not None:
                        assert model_results is not None
                        parse_model(lines[gene_start:line_num], model_results)
                        genes[current_gene - 1] = model_results
                    gene_start = line_num
                    current_gene = int(gene_res.group(1))
                    model_results = {'description': siteclass_model}
            if len(genes[current_gene - 1]) == 0:
                model_results = parse_model(lines[gene_start:], model_results)
                genes[current_gene - 1] = model_results
        else:
            model_results = {'description': siteclass_model}
            model_results = parse_model(lines, model_results)
            ns_sites[current_model] = model_results
    else:
        current_model = None
        model_start = None
        for line_num, line in enumerate(lines):
            model_res = model_re.match(line)
            if model_res:
                if current_model is not None:
                    parse_model(lines[model_start:line_num], model_results)
                    ns_sites[current_model] = model_results
                model_start = line_num
                current_model = int(model_res.group(1))
                model_results = {'description': model_res.group(2)}
        if ns_sites.get(current_model) is None:
            model_results = parse_model(lines[model_start:], model_results)
            ns_sites[current_model] = model_results
    if len(ns_sites) == 1:
        m0 = ns_sites.get(0)
        if not m0 or len(m0) > 1:
            results['NSsites'] = ns_sites
    elif len(ns_sites) > 1:
        results['NSsites'] = ns_sites
    return results