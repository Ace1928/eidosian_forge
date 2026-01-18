import sys
from unittest import mock
from libcloud.test import unittest
from libcloud.compute.drivers.libvirt_driver import LibvirtNodeDriver, have_libvirt
def test_bad_map(self, *args, **keywargs):
    driver = LibvirtNodeDriver('')
    arp_table = driver._parse_ip_table_neigh(self.bad_output_str)
    self.assertEqual(len(arp_table), 2)
    arp_table = driver._parse_ip_table_neigh(self.arp_output_str)
    self.assertEqual(len(arp_table), 0)