import re
def parse_parameters(lines, results, num_params):
    """Parse the various parameters from the file."""
    parameters = {}
    parameters = parse_parameter_list(lines, parameters, num_params)
    parameters = parse_kappas(lines, parameters)
    parameters = parse_rates(lines, parameters)
    parameters = parse_freqs(lines, parameters)
    results['parameters'] = parameters
    return results