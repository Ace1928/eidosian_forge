import codecs
def punycode_encode(text):
    base, extended = segregate(text)
    deltas = insertion_unsort(text, extended)
    extended = generate_integers(len(base), deltas)
    if base:
        return base + b'-' + extended
    return extended