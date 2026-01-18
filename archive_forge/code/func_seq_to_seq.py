import collections
import keras.src as keras
def seq_to_seq():
    """Sequence to sequence model."""
    num_encoder_tokens = 3
    num_decoder_tokens = 3
    latent_dim = 2
    encoder_inputs = keras.Input(shape=(None, num_encoder_tokens))
    encoder = keras.layers.LSTM(latent_dim, return_state=True)
    _, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    decoder_inputs = keras.Input(shape=(None, num_decoder_tokens))
    decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = keras.layers.Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return ModelFn(model, [(None, 2, num_encoder_tokens), (None, 2, num_decoder_tokens)], (None, 2, num_decoder_tokens))