import hashlib
import sys
from pyasn1.codec.der import decoder
from pyasn1.codec.der import encoder
from pyasn1.type import univ
from pyasn1_modules import rfc2560
from pyasn1_modules import rfc2459
from pyasn1_modules import pem
def mkOcspRequest(issuerCert, userCert):
    issuerTbsCertificate = issuerCert.getComponentByName('tbsCertificate')
    issuerSubject = issuerTbsCertificate.getComponentByName('subject')
    userTbsCertificate = userCert.getComponentByName('tbsCertificate')
    userIssuer = userTbsCertificate.getComponentByName('issuer')
    assert issuerSubject == userIssuer, '%s\n%s' % (issuerSubject.prettyPrint(), userIssuer.prettyPrint())
    userIssuerHash = hashlib.sha1(encoder.encode(userIssuer)).digest()
    issuerSubjectPublicKey = issuerTbsCertificate.getComponentByName('subjectPublicKeyInfo').getComponentByName('subjectPublicKey')
    issuerKeyHash = hashlib.sha1(valueOnlyBitStringEncoder(issuerSubjectPublicKey)).digest()
    userSerialNumber = userTbsCertificate.getComponentByName('serialNumber')
    request = rfc2560.Request()
    reqCert = request.setComponentByName('reqCert').getComponentByName('reqCert')
    hashAlgorithm = reqCert.setComponentByName('hashAlgorithm').getComponentByName('hashAlgorithm')
    hashAlgorithm.setComponentByName('algorithm', sha1oid)
    reqCert.setComponentByName('issuerNameHash', userIssuerHash)
    reqCert.setComponentByName('issuerKeyHash', issuerKeyHash)
    reqCert.setComponentByName('serialNumber', userSerialNumber)
    ocspRequest = rfc2560.OCSPRequest()
    tbsRequest = ocspRequest.setComponentByName('tbsRequest').getComponentByName('tbsRequest')
    tbsRequest.setComponentByName('version', 'v1')
    requestList = tbsRequest.setComponentByName('requestList').getComponentByName('requestList')
    requestList.setComponentByPosition(0, request)
    return ocspRequest