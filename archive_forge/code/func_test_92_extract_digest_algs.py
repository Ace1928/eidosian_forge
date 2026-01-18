import logging
import warnings
from passlib import hash
from passlib.utils.compat import u
from passlib.tests.utils import TestCase, HandlerCase
from passlib.tests.test_handlers import UPASS_WAV
def test_92_extract_digest_algs(self):
    """test scram.extract_digest_algs()"""
    eda = self.handler.extract_digest_algs
    self.assertEqual(eda('$scram$4096$QSXCR.Q6sek8bf92$sha-1=HZbuOlKbWl.eR8AfIposuKbhX30'), ['sha-1'])
    self.assertEqual(eda('$scram$4096$QSXCR.Q6sek8bf92$sha-1=HZbuOlKbWl.eR8AfIposuKbhX30', format='hashlib'), ['sha1'])
    self.assertEqual(eda('$scram$4096$QSXCR.Q6sek8bf92$sha-1=HZbuOlKbWl.eR8AfIposuKbhX30,sha-256=qXUXrlcvnaxxWG00DdRgVioR2gnUpuX5r.3EZ1rdhVY,sha-512=lzgniLFcvglRLS0gt.C4gy.NurS3OIOVRAU1zZOV4P.qFiVFO2/edGQSu/kD1LwdX0SNV/KsPdHSwEl5qRTuZQ'), ['sha-1', 'sha-256', 'sha-512'])