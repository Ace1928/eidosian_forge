from __future__ import absolute_import, division, print_function
import re
import time
import xml.etree.ElementTree
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import iControlRestSession
from ..module_utils.teem import send_teem
@property
def license_envelope(self):
    result = '<?xml version="1.0" encoding="UTF-8"?>\n        <SOAP-ENV:Envelope xmlns:ns3="http://www.w3.org/2001/XMLSchema"\n                           xmlns:SOAP-ENC="http://schemas.xmlsoap.org/soap/encoding/"\n                           xmlns:ns0="http://schemas.xmlsoap.org/soap/encoding/"\n                           xmlns:ns1="https://{0}/license/services/urn:com.f5.license.v5b.ActivationService"\n                           xmlns:ns2="http://schemas.xmlsoap.org/soap/envelope/"\n                           xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n                           xmlns:SOAP-ENV="http://schemas.xmlsoap.org/soap/envelope/"\n                           SOAP-ENV:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">\n          <SOAP-ENV:Header/>\n          <ns2:Body>\n            <ns1:getLicense>\n              <dossier xsi:type="ns3:string">{1}</dossier>\n              <eula xsi:type="ns3:string">{eula}</eula>\n              <email xsi:type="ns3:string">{email}</email>\n              <firstName xsi:type="ns3:string">{first_name}</firstName>\n              <lastName xsi:type="ns3:string">{last_name}</lastName>\n              <companyName xsi:type="ns3:string">{company}</companyName>\n              <phone xsi:type="ns3:string">{phone}</phone>\n              <jobTitle xsi:type="ns3:string">{job_title}</jobTitle>\n              <address xsi:type="ns3:string">{address}</address>\n              <city xsi:type="ns3:string">{city}</city>\n              <stateProvince xsi:type="ns3:string">{state}</stateProvince>\n              <postalCode xsi:type="ns3:string">{postal_code}</postalCode>\n              <country xsi:type="ns3:string">{country}</country>\n            </ns1:getLicense>\n          </ns2:Body>\n        </SOAP-ENV:Envelope>'
    result = result.format(self.license_server, self.dossier, **self.license_options)
    return result