import gettext
import logging
import os
import sqlite3
import sys
@classmethod
def by_alpha_3(cls, code):
    return Country.get_country(code, 'alpha_3')