import json
import logging
from bs4 import BeautifulSoup
from mechanize import ParseResponseEx
from mechanize._form import AmbiguityError
from mechanize._form import ControlNotFoundError
from mechanize._form import ListControl
from urlparse import urlparse
def pick_interaction(self, _base='', content='', req=None):
    logger.info('pick_interaction baseurl: %s', _base)
    unic = content
    if content:
        _bs = BeautifulSoup(content)
    else:
        _bs = None
    for interaction in self.interactions:
        _match = 0
        for attr, val in interaction['matches'].items():
            if attr == 'url':
                logger.info('matching baseurl against: %s', val)
                if val == _base:
                    _match += 1
            elif attr == 'title':
                logger.info("matching '%s' against title", val)
                if _bs is None:
                    break
                if _bs.title is None:
                    break
                if val in _bs.title.contents:
                    _match += 1
                else:
                    _c = _bs.title.contents
                    if isinstance(_c, list) and (not isinstance(_c, str)):
                        for _line in _c:
                            if val in _line:
                                _match += 1
                                continue
            elif attr == 'content':
                if unic and val in unic:
                    _match += 1
            elif attr == 'class':
                if req and val == req:
                    _match += 1
        if _match == len(interaction['matches']):
            logger.info('Matched: %s', interaction['matches'])
            return interaction
    raise InteractionNeeded('No interaction matched')