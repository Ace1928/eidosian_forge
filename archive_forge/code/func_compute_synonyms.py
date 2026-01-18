from nltk.corpus import wordnet
from nltk.wsd import lesk
import spacy
def compute_synonyms(parsed_sentences, words_to_disambiguate_per_sentence, window_size=10, disambiguate=True):
    final_parsed_sentences = []
    words_before_after = window_size // 2
    for parsed_sentence, words_to_disambiguate in zip(parsed_sentences, words_to_disambiguate_per_sentence):
        words = [{'text': word['text'], 'lemma': word['lemma'], 'pos': mapping_pos_spacy_pos_wordnet.get(word['pos'], list(mapping_pos_spacy_pos_wordnet.values()))} for word in parsed_sentence]
        tokenized_sentence = []
        if disambiguate:
            tokenized_sentence = [word_attributes['text'] for word_attributes in words]
        for word_pos, word_attributes in enumerate(words):
            word_attributes['synonyms'] = {}
            if word_attributes['text'] in words_to_disambiguate or word_attributes['lemma'] in words_to_disambiguate:
                word_attributes['synonyms'] = wordnet.synsets(word_attributes['lemma'])
                if len(word_attributes['synonyms']) == 0:
                    word_attributes['synonyms'] = None
                if disambiguate:
                    context_tokens = tokenized_sentence[max(word_pos - words_before_after, 0):word_pos] + tokenized_sentence[word_pos + 1:word_pos + 1 + words_before_after]
                    word_attributes['synonyms'] = lesk(context_tokens, word_attributes['lemma'], word_attributes['pos'], word_attributes['synonyms'])
                if isinstance(word_attributes['synonyms'], list):
                    word_attributes['synonyms'] = word_attributes['synonyms'][0]
                if word_attributes['synonyms'] is not None:
                    word_attributes['synonyms'] = {synonym.replace('_', ' ') for synonym in word_attributes['synonyms'].lemma_names()}
                    word_attributes['synonyms'].discard(word_attributes['text'])
                    word_attributes['synonyms'].discard(word_attributes['lemma'])
                else:
                    word_attributes['synonyms'] = {}
        final_parsed_sentences.append(words)
    return final_parsed_sentences