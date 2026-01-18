from __future__ import (absolute_import, division, print_function)
import json
import zipfile
import io
def import_certificate_into_primary(self, primary_node):
    cert_id = self.return_id_of_certificate()
    data = json.dumps({'id': cert_id, 'export': 'CERTIFICATE'})
    url = 'https://{ip}/api/v1/certs/system-certificate/export'.format(ip=self.ip)
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    try:
        response = requests.post(url=url, timeout=15, data=data, headers=headers, auth=(self.username, self.password), verify=False)
    except Exception as e:
        raise AnsibleActionFail(e)
    if not response.status_code == 200:
        raise AnsibleActionFail('Received status code {status_code} when exporting certificate.'.format(status_code=str(response.status_code)))
    zf = zipfile.ZipFile(io.BytesIO(response.content), 'r')
    cert_data = zf.read('Defaultselfsignedservercerti.pem')
    data = json.dumps({'allowBasicConstraintCAFalse': True, 'allowOutOfDateCert': False, 'allowSHA1Certificates': True, 'trustForCertificateBasedAdminAuth': True, 'trustForCiscoServicesAuth': True, 'trustForClientAuth': True, 'data': cert_data.decode('utf-8'), 'trustForIseAuth': True, 'name': self.name, 'validateCertificateExtensions': True})
    url = 'https://{primary_ip}/api/v1/certs/trusted-certificate/import'.format(primary_ip=primary_node.ip)
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    try:
        response = requests.post(url=url, timeout=15, data=data, headers=headers, auth=(self.primary_node.username, self.primary_node.password), verify=False)
        return_message = json.loads(response.text)['response']['message']
    except Exception as e:
        raise AnsibleActionFail(e)
    if not response.status_code == 200:
        if not (return_message == 'Trust certificate was added successfully' or return_message == 'Certificates are having same subject, same serial number and they are binary equal. Hence skipping the replace'):
            raise AnsibleActionFail('Unexpected response from API. Received response was {message}'.format(message=return_message))