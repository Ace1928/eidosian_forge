import copy
def entity_groups(self):
    """
        Group consecutive entity tokens with the same NER tag.
        """
    entities = self.entities()
    if not entities:
        return None
    non_ent = self.opts.get('non_ent', 'O')
    groups = []
    idx = 0
    while idx < len(entities):
        ner_tag = entities[idx]
        if ner_tag != non_ent:
            start = idx
            while idx < len(entities) and entities[idx] == ner_tag:
                idx += 1
            groups.append((self.slice(start, idx).untokenize(), ner_tag))
        else:
            idx += 1
    return groups