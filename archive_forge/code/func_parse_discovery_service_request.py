from urllib import parse
from saml2.entity import Entity
from saml2.response import VerificationError
def parse_discovery_service_request(self, url='', query=''):
    if url:
        part = parse.urlparse(url)
        dsr = parse.parse_qs(part[4])
    elif query:
        dsr = parse.parse_qs(query)
    else:
        dsr = {}
    for key in ['isPassive', 'return', 'returnIDParam', 'policy', 'entityID']:
        try:
            if len(dsr[key]) != 1:
                raise Exception(f'Invalid DS request keys: {key}')
            dsr[key] = dsr[key][0]
        except KeyError:
            pass
    if 'return' in dsr:
        part = parse.urlparse(dsr['return'])
        if part.query:
            qp = parse.parse_qs(part.query)
            if 'returnIDParam' in dsr:
                if dsr['returnIDParam'] in qp.keys():
                    raise Exception('returnIDParam value should not be in the query params')
            elif 'entityID' in qp.keys():
                raise Exception('entityID should not be in the query params')
    else:
        raise VerificationError("Missing mandatory parameter 'return'")
    if 'policy' not in dsr:
        dsr['policy'] = IDPDISC_POLICY
    is_passive = dsr.get('isPassive')
    if is_passive not in ['true', 'false']:
        raise ValueError(f"Invalid value '{is_passive}' for attribute 'isPassive'")
    if 'isPassive' in dsr and dsr['isPassive'] == 'true':
        dsr['isPassive'] = True
    else:
        dsr['isPassive'] = False
    if 'returnIDParam' not in dsr:
        dsr['returnIDParam'] = 'entityID'
    return dsr