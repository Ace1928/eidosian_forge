import base64
from unittest import TestCase
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
import nacl.public
import six
from macaroonbakery.bakery import _codec as codec
def test_decode_caveat_v1_from_go(self):
    tp_key = bakery.PrivateKey(nacl.public.PrivateKey(base64.b64decode('TSpvLpQkRj+T3JXnsW2n43n5zP/0X4zn0RvDiWC3IJ0=')))
    fp_key = bakery.PrivateKey(nacl.public.PrivateKey(base64.b64decode('KXpsoJ9ujZYi/O2Cca6kaWh65MSawzy79LWkrjOfzcs=')))
    root_key = base64.b64decode('vDxEmWZEkgiNEFlJ+8ruXe3qDSLf1H+o')
    encrypted_cav = six.b('eyJUaGlyZFBhcnR5UHVibGljS2V5IjoiOFA3R1ZZc3BlWlN4c3hFdmJsSVFFSTFqdTBTSWl0WlIrRFdhWE40cmxocz0iLCJGaXJzdFBhcnR5UHVibGljS2V5IjoiSDlqSFJqSUxidXppa1VKd2o5VGtDWk9qeW5oVmtTdHVsaUFRT2d6Y0NoZz0iLCJOb25jZSI6Ii9lWTRTTWR6TGFxbDlsRFc3bHUyZTZuSzJnVG9veVl0IiwiSWQiOiJra0ZuOGJEaEt4RUxtUjd0NkJxTU0vdHhMMFVqaEZjR1BORldUUExGdjVla1dWUjA4Uk1sbGJhc3c4VGdFbkhzM0laeVo0V2lEOHhRUWdjU3ljOHY4eUt4dEhxejVEczJOYmh1ZDJhUFdtUTVMcVlNWitmZ2FNaTAxdE9DIn0=')
    cav = bakery.decode_caveat(tp_key, encrypted_cav)
    self.assertEqual(cav, bakery.ThirdPartyCaveatInfo(condition='caveat condition', first_party_public_key=fp_key.public_key, third_party_key_pair=tp_key, root_key=root_key, caveat=encrypted_cav, version=bakery.VERSION_1, id=None, namespace=bakery.legacy_namespace()))