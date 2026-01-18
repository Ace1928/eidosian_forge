import sys
from pyasn1.codec.der import decoder as der_decoder
from pyasn1.codec.der import encoder as der_encoder
from pyasn1_modules import pem
from pyasn1_modules import rfc2315
def testDerCodecDecodeOpenTypes(self):
    substrate = pem.readBase64fromText(self.pem_text_reordered)
    asn1Object, rest = der_decoder.decode(substrate, asn1Spec=self.asn1Spec, decodeOpenTypes=True)
    assert not rest
    assert asn1Object.prettyPrint()
    assert der_encoder.encode(asn1Object) == substrate