import json
import requests
def get_grafana_releases():
    r = requests.get('https://api.github.com/repos/grafana/grafana/releases?per_page=50', headers={'Accept': 'application/vnd.github.v3+json'})
    if r.status_code != 200:
        raise Exception('Failed to get releases from GitHub')
    return r.json()