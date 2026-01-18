from . import matrix
def evaluate_word(identity_matrix, generator_matrices, inverse_matrices, word, G):
    m = identity_matrix
    image_of_word = _apply_hom_to_word(word, G)
    for letter in image_of_word:
        if letter.isupper():
            g = inverse_matrices[ord(letter) - ord('A')]
        else:
            g = generator_matrices[ord(letter) - ord('a')]
        m = matrix.matrix_mult(m, g)
    return m