import os
import pickle
import re
from xml.etree import ElementTree as ET
from nltk.tag import ClassifierBasedTagger, pos_tag
from nltk.chunk.api import ChunkParserI
from nltk.chunk.util import ChunkScore
from nltk.data import find
from nltk.tokenize import word_tokenize
from nltk.tree import Tree
def load_ace_file(textfile, fmt):
    print(f'  - {os.path.split(textfile)[1]}')
    annfile = textfile + '.tmx.rdc.xml'
    entities = []
    with open(annfile) as infile:
        xml = ET.parse(infile).getroot()
    for entity in xml.findall('document/entity'):
        typ = entity.find('entity_type').text
        for mention in entity.findall('entity_mention'):
            if mention.get('TYPE') != 'NAME':
                continue
            s = int(mention.find('head/charseq/start').text)
            e = int(mention.find('head/charseq/end').text) + 1
            entities.append((s, e, typ))
    with open(textfile) as infile:
        text = infile.read()
    text = re.sub('<(?!/?TEXT)[^>]+>', '', text)

    def subfunc(m):
        return ' ' * (m.end() - m.start() - 6)
    text = re.sub('[\\s\\S]*<TEXT>', subfunc, text)
    text = re.sub('</TEXT>[\\s\\S]*', '', text)
    text = re.sub('``', ' "', text)
    text = re.sub("''", '" ', text)
    entity_types = {typ for s, e, typ in entities}
    if fmt == 'binary':
        i = 0
        toks = Tree('S', [])
        for s, e, typ in sorted(entities):
            if s < i:
                s = i
            if e <= s:
                continue
            toks.extend(word_tokenize(text[i:s]))
            toks.append(Tree('NE', text[s:e].split()))
            i = e
        toks.extend(word_tokenize(text[i:]))
        yield toks
    elif fmt == 'multiclass':
        i = 0
        toks = Tree('S', [])
        for s, e, typ in sorted(entities):
            if s < i:
                s = i
            if e <= s:
                continue
            toks.extend(word_tokenize(text[i:s]))
            toks.append(Tree(typ, text[s:e].split()))
            i = e
        toks.extend(word_tokenize(text[i:]))
        yield toks
    else:
        raise ValueError('bad fmt value')