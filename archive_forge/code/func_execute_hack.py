import re, sys, os, tempfile, json
def execute_hack():
    data = json.loads(sys.argv[1].decode('base64'))
    if data['backend'] == 'cyphc':
        solver = phc_direct_base
    elif data['backend'] == 'phcpy':
        solver = phcpy_direct_base
    else:
        raise ValueError('nonexistent backend specified')
    sols = [serialize_sol_dict(sol) for sol in solver(data['vars'], data['polys'], **data['kwargs'])]
    sys.stdout.write(json.dumps(sols))