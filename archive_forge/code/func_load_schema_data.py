def load_schema_data():
    import os.path
    import json
    json_file = os.path.join(os.path.dirname(__file__), 'debugProtocol.json')
    if not os.path.exists(json_file):
        import requests
        req = requests.get('https://raw.githubusercontent.com/microsoft/debug-adapter-protocol/gh-pages/debugAdapterProtocol.json')
        assert req.status_code == 200
        with open(json_file, 'wb') as stream:
            stream.write(req.content)
    with open(json_file, 'rb') as json_contents:
        json_schema_data = json.loads(json_contents.read())
    return json_schema_data