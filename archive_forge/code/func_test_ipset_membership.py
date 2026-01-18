import pickle
import sys
import weakref
import pytest
from netaddr import IPAddress, IPNetwork, IPRange, IPSet, cidr_exclude
def test_ipset_membership():
    iprange = IPRange('192.0.1.255', '192.0.2.16')
    assert iprange.cidrs() == [IPNetwork('192.0.1.255/32'), IPNetwork('192.0.2.0/28'), IPNetwork('192.0.2.16/32')]
    ipset = IPSet(['192.0.2.0/28'])
    assert [(str(ip), ip in ipset) for ip in iprange] == [('192.0.1.255', False), ('192.0.2.0', True), ('192.0.2.1', True), ('192.0.2.2', True), ('192.0.2.3', True), ('192.0.2.4', True), ('192.0.2.5', True), ('192.0.2.6', True), ('192.0.2.7', True), ('192.0.2.8', True), ('192.0.2.9', True), ('192.0.2.10', True), ('192.0.2.11', True), ('192.0.2.12', True), ('192.0.2.13', True), ('192.0.2.14', True), ('192.0.2.15', True), ('192.0.2.16', False)]