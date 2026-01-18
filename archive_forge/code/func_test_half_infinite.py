import unittest
from traits.api import (
from traits.testing.optional_dependencies import numpy, requires_numpy
def test_half_infinite(self):
    ice_temperatures = [-273.15, -273.0, -100.0, -1.0, -0.1, -0.001]
    water_temperatures = [0.001, 0.1, 1.0, 50.0, 99.0, 99.9, 99.999]
    steam_temperatures = [100.001, 100.1, 101.0, 1000.0, 1e+100]
    for temperature in steam_temperatures:
        self.model.steam_temperature = temperature
        self.assertEqual(self.model.steam_temperature, temperature)
    for temperature in ice_temperatures + water_temperatures:
        self.model.steam_temperature = 1729.0
        with self.assertRaises(TraitError):
            self.model.steam_temperature = temperature
        self.assertEqual(self.model.steam_temperature, 1729.0)
    for temperature in ice_temperatures:
        self.model.ice_temperature = temperature
        self.assertEqual(self.model.ice_temperature, temperature)
    for temperature in water_temperatures + steam_temperatures:
        self.model.ice_temperature = -1729.0
        with self.assertRaises(TraitError):
            self.model.ice_temperature = temperature
        self.assertEqual(self.model.ice_temperature, -1729.0)