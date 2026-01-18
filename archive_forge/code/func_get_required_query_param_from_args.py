from urllib import parse as urlparse
def get_required_query_param_from_args(required_traits, forbidden_traits):
    required_query_params = []
    and_traits = []
    for required in required_traits:
        if ',' in required:
            required_query_params.append('in:' + required)
        else:
            and_traits.append(required)
    and_query = ','.join(and_traits + forbidden_traits)
    if and_query:
        required_query_params.append(and_query)
    return required_query_params