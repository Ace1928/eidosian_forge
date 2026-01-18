import pytest
def test_de_tokenizer_handles_long_text(de_tokenizer):
    text = 'Die Verwandlung\n\nAls Gregor Samsa eines Morgens aus unruhigen Träumen erwachte, fand er sich in\nseinem Bett zu einem ungeheueren Ungeziefer verwandelt.\n\nEr lag auf seinem panzerartig harten Rücken und sah, wenn er den Kopf ein wenig\nhob, seinen gewölbten, braunen, von bogenförmigen Versteifungen geteilten\nBauch, auf dessen Höhe sich die Bettdecke, zum gänzlichen Niedergleiten bereit,\nkaum noch erhalten konnte. Seine vielen, im Vergleich zu seinem sonstigen\nUmfang kläglich dünnen Beine flimmerten ihm hilflos vor den Augen.\n\n»Was ist mit mir geschehen?«, dachte er.'
    tokens = de_tokenizer(text)
    assert len(tokens) == 109