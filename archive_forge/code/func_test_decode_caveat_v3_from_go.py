import base64
from unittest import TestCase
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
import nacl.public
import six
from macaroonbakery.bakery import _codec as codec
def test_decode_caveat_v3_from_go(self):
    tp_key = bakery.PrivateKey(nacl.public.PrivateKey(base64.b64decode('TSpvLpQkRj+T3JXnsW2n43n5zP/0X4zn0RvDiWC3IJ0=')))
    fp_key = bakery.PrivateKey(nacl.public.PrivateKey(base64.b64decode('KXpsoJ9ujZYi/O2Cca6kaWh65MSawzy79LWkrjOfzcs=')))
    root_key = base64.b64decode(b'oqOXI3/Mz/pKjCuFOt2eYxb7ndLq66GY')
    encrypted_cav = bakery.b64decode('A_D-xlUf2MdGMgtu7OKRQnCP1OQJk6PKeFWRK26WIBA6DNwKGNLeFSkD2M-8AEYvmgVH95GWu7T7caKxKhhOQFcEKgnXKJvYXxz1zin4cZc4Q6C7gVqA-J4_j31LX4VKxymqG62UGPo78wOv0_fKjr3OI6PPJOYOQgBMclemlRF2')
    cav = bakery.decode_caveat(tp_key, encrypted_cav)
    self.assertEqual(cav, bakery.ThirdPartyCaveatInfo(condition='third party condition', first_party_public_key=fp_key.public_key, third_party_key_pair=tp_key, root_key=root_key, caveat=encrypted_cav, version=bakery.VERSION_3, id=None, namespace=bakery.legacy_namespace()))