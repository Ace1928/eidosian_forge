import unittest
from tempfile import mkdtemp
from shutil import rmtree
def test_collision(self):
    wid = self.root
    self.assertEqual(wid.pos, [0, 0])
    self.assertEqual(wid.size, [100, 100])
    self.assertEqual(wid.collide_point(-1, -1), False)
    self.assertEqual(wid.collide_point(0, 0), True)
    self.assertEqual(wid.collide_point(50, 50), True)
    self.assertEqual(wid.collide_point(100, 100), True)
    self.assertEqual(wid.collide_point(200, 0), False)
    self.assertEqual(wid.collide_point(500, 500), False)