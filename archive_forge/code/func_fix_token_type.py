import json
def fix_token_type(r):
    token = json.loads(r.text)
    token.setdefault('token_type', 'Bearer')
    fixed_token = json.dumps(token)
    r._content = fixed_token.encode()
    return r